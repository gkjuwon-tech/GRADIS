#!/usr/bin/env python3
"""
GIF desaturation tool — reduces color saturation of an animated GIF.
Usage:
  python tools/desaturate_gif.py input.gif output.gif --factor 0.35
"""
import argparse
from PIL import Image, ImageEnhance, ImageSequence

def desaturate_gif(input_path, output_path, factor=0.35):
    print(f"Reading GIF from: {input_path}")
    img = Image.open(input_path)
    
    frames = []
    durations = []
    default_duration = img.info.get('duration', 100)
    
    for frame in ImageSequence.Iterator(img):
        # Convert to RGB to apply color enhancement
        frame_rgb = frame.convert('RGB')
        
        # Apply saturation factor (0.0 means black-and-white, 1.0 means original)
        enhancer = ImageEnhance.Color(frame_rgb)
        enhanced = enhancer.enhance(factor)
        
        # Convert back to Palette mode for GIF format
        frame_p = enhanced.convert('P', palette=Image.ADAPTIVE)
        
        frames.append(frame_p)
        durations.append(frame.info.get('duration', default_duration))
        
    if not frames:
        raise ValueError("No frames found in input GIF.")
        
    print(f"Saving desaturated GIF ({len(frames)} frames) to: {output_path} (factor={factor})")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=durations,
        loop=0
    )
    print("Done!")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to input GIF")
    ap.add_argument("output", help="Path to output GIF")
    ap.add_argument("--factor", type=float, default=0.35, help="Saturation factor (0.0 to 1.0)")
    args = ap.parse_args()
    
    desaturate_gif(args.input, args.output, args.factor)

if __name__ == "__main__":
    main()
