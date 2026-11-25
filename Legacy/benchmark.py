"""Configurable Windows/CUDA feasibility benchmark; uses the existing HF cache."""
import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
from PIL import Image, ImageOps
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)

GIB = 1024 ** 3


class MemoryMonitor:
    # Sample process working set and system pressure, including model loading.
    def __init__(self):
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak_process_ram = 0
        self.minimum_available_ram = float('inf')
        self.peak_system_ram_percent = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            memory = psutil.virtual_memory()
            self.peak_process_ram = max(
                self.peak_process_ram, self.process.memory_info().rss
            )
            self.minimum_available_ram = min(
                self.minimum_available_ram, memory.available
            )
            self.peak_system_ram_percent = max(
                self.peak_system_ram_percent, memory.percent
            )
            self.stop_event.wait(0.1)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        return {
            'sample_interval_seconds': 0.1,
            'peak_process_working_set_gib': self.peak_process_ram / GIB,
            'minimum_system_available_ram_gib': self.minimum_available_ram / GIB,
            'peak_system_ram_percent': self.peak_system_ram_percent,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of edits, including one warm-up (minimum 2).')
    parser.add_argument('--input', type=Path, default=Path('pix2pix-results/input.png'))
    parser.add_argument('--offload', action='store_true',
                        help='Test CPU offloading separately if GPU-only fails.')
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('--runs must be at least 2 to include a warm-up and a timed run.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable in this Python environment.')

    with Image.open(args.input) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    width, height = image.width // 8 * 8, image.height // 8 * 8
    if min(width, height) < 8:
        raise SystemExit('Image is too small.')
    image = image.crop((0, 0, width, height))
    output = Path('benchmarks') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    image.save(output / 'input.png')

    prompt = 'What would it look like if it was snowing?'
    report = {
        'model': 'timbrooks/instruct-pix2pix',
        'torch_version': torch.__version__,
        'cuda_build': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'gpu_total_gib': torch.cuda.get_device_properties(0).total_memory / GIB,
        'system_total_ram_gib': psutil.virtual_memory().total / GIB,
        'mode': 'cpu_offload' if args.offload else 'gpu_only',
        'input_path': str(args.input.resolve()),
        'input_size': image.size,
        'prompt': prompt,
        'steps': 30,
        'guidance_scale': 7.5,
        'image_guidance_scale': 1.5,
        'runs': [],
        'requested_runs': args.runs,
    }
    monitor = MemoryMonitor()
    monitor.start()
    stage = 'model_loading'
    try:
        print('Loading model from cache; missing files may download...', flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            report['model'], torch_dtype=torch.float16,
            use_safetensors=True, low_cpu_mem_usage=True,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        if args.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to('cuda')
        torch.cuda.synchronize()
        report['load_seconds_including_any_download'] = time.perf_counter() - started
        report['load_peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / GIB
        report['load_peak_gpu_reserved_gib'] = torch.cuda.max_memory_reserved() / GIB
        print(f"Loaded in {report['load_seconds_including_any_download']:.1f}s")

        for index in range(args.runs):
            stage = f'inference_{index + 1}'
            seed = 42 + index
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt, image=image, num_inference_steps=30,
                    guidance_scale=7.5, image_guidance_scale=1.5,
                    generator=torch.Generator(device='cpu').manual_seed(seed),
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            filename = f'edit_{index + 1:02d}.png'
            result.images[0].save(output / filename)
            flags = result.nsfw_content_detected
            report['runs'].append({
                'run': index + 1, 'warmup': index == 0, 'seed': seed,
                'inference_seconds': elapsed,
                'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / GIB,
                'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved() / GIB,
                'output_size': result.images[0].size,
                'safety_flags': None if flags is None else [bool(x) for x in flags],
                'output_file': filename,
            })
            print(f'Run {index + 1}/{args.runs}: {elapsed:.1f}s', flush=True)
            del result

        stable = [row['inference_seconds'] for row in report['runs'][1:]]
        average = sum(stable) / len(stable)
        report['mean_seconds_excluding_warmup'] = average
        report['estimated_8_hour_outputs_excluding_io_and_failures'] = int(8 * 3600 / average)
        report['status'] = 'completed'
    except Exception as error:
        report['status'] = 'failed'
        report['failed_stage'] = stage
        report['error'] = f'{type(error).__name__}: {error}'
        print(report['error'], flush=True)
    finally:
        report['sampled_ram_whole_benchmark'] = monitor.finish()
        destination = output / 'benchmark.json'
        destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report: {destination.resolve()}')
        print(json.dumps(report, indent=2))
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
