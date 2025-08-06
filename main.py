import os
import argparse

from lib.pgd import PGD

def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate targeted adversarial example using projected gradient descent.")
    parser.add_argument('--image_path', required=True, help="input image path (.png, .jpg, or .jpeg).")
    parser.add_argument('--target_class', type=int, required=True, help="target ImageNet class (0-999).")
    parser.add_argument('--output_path', default='./test_out', help="output path.")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None, help="device selection")
    parser.add_argument('--use_l2', action='store_true', help='Use L2 norm for projected gradient descent, may take longer.')
    args = parser.parse_args()

    # Input validations
    assert args.target_class >=0 and args.target_class <1000, "Target ImageNet class should be between 0 and 999. See https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/ for more details."
    assert os.path.isfile(args.image_path) and args.image_path.lower().endswith(('.png', '.jpg', '.jpeg')), "Path to image file is not valid."
    os.makedirs(args.output_path, exist_ok=True)

    # Generate adversarial example using projected gradient descent
    pgd = PGD(args.use_l2, args.device)
    success = pgd.generate_from_file(args.image_path, args.target_class, args.output_path)

    if success:
        print(f"Successfully generated targeted adversarial example in {args.output_path}...")
    else:
        print(f"Failed to generate targeted adversarial example for {args.image_path}...")

if __name__ == '__main__':
    main()