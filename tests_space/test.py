from __future__ import print_function
import glob
import sys
import cv2 as cv
import argparse

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

def main():
    parser = argparse.ArgumentParser(
        prog='test.py',
        description='Stitching sample.'
    )

    parser.add_argument(
        '--mode',
        type=int,
        choices=modes,
        default=cv.Stitcher_PANORAMA,
        help=(
            'Stitcher mode. PANORAMA (%d) for general panoramas, '
            'SCANS (%d) for flat/affine surfaces like scanned images.'
            % modes
        )
    )

    parser.add_argument(
        '--output',
        default='result.jpg',
        help='Output stitched image filename.'
    )

    args = parser.parse_args()

    # ------------------------------
    # Load images from folder
    # ------------------------------
    imgs = []
    image_paths = sorted(glob.glob('SampleImg/Slides/*.JPG'))

    if len(image_paths) == 0:
        print("No images found in PanoImg/ folder")
        sys.exit(-1)

    for path in image_paths:
        img = cv.imread(path)
        if img is not None:
            imgs.append(img)

    if len(imgs) < 2:
        print("Need at least 2 images to stitch.")
        sys.exit(-1)

    # ------------------------------
    # Create Stitcher
    # ------------------------------
    stitcher = cv.Stitcher.create(args.mode)

    print("Stitching started...")
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)

    # ------------------------------
    # Save output
    # ------------------------------
    cv.imshow("result",pano)
    cv.imwrite(args.output, pano)
    print("Stitching completed. Saved as:", args.output)

    print("Done")

# Run main
if __name__ == "__main__":
    main()
    cv.destroyAllWindows()
