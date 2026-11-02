# Figure sources

Product images downloaded 2026-10-05 for the inspection report. All are
manufacturer-owned marketing/datasheet assets; add attribution in the report
(e.g. "Image: Universal Robots A/S", "Image: OnRobot A/S", "Image: RealSense").

## ur5e.png — Universal Robots UR5e

- File: `ur5e.png`, 1600 x 1200 px, PNG (opaque), 630 KB
- Direct image URL (UR's Storyblok CDN, resized by the CDN to 1600 px wide):
  https://a.storyblok.com/f/169662/5873x4405/3a94f66de5/ur5e-4-backgroundwarm50.png/m/1600x0
  - Original full-resolution asset (5873 x 4405): drop the `/m/1600x0` suffix.
- Source page: the former UR5e product page
  https://www.universal-robots.com/products/ur5e/ — this URL now redirects to
  https://www.universal-robots.com/products/ (UR renamed the UR5e to UR7e in
  2025 and retired the UR5e page). The image URL was recovered from the
  Wayback Machine snapshot of the page (2024-12-20):
  https://web.archive.org/web/20241220075421/https://www.universal-robots.com/products/ur5e/
  The CDN asset itself is still live and served by universal-robots.com's CDN.
- Copyright: Universal Robots A/S.

### ur5e_datasheet_cutout.png — alternate UR5e render (transparent background)

- File: `ur5e_datasheet_cutout.png`, 1067 x 1219 px, PNG with alpha, 365 KB
- Extracted (image + soft mask, recombined with PIL) from page 1 of the
  official UR5e technical datasheet:
  https://www.universal-robots.com/media/1807465/ur5e_e-series_datasheets_web.pdf
  ("UR5e Technical Specification", updated December 2024, (c) 2024 Universal
  Robots A/S).
- Use this one if the figure must sit on pure white / needs a cutout.

## rg2.jpg — OnRobot RG2 gripper

- File: `rg2.jpg`, 1000 x 1000 px, JPEG, white background, 77 KB
- Direct image URL (OnRobot B2B webshop CDN):
  https://b2b.onrobot.com/imagesv2/onrobot/trim:white/1000/4155be0d-7d34-481e-a342-213b2490425f/102012+-+RG2+1000x1000px.jpg
- Source page: https://b2b.onrobot.com/rg2/ (OnRobot's own B2B shop, SKU 102012).
- Main product page (same gripper, used for the gallery/hero images, transparent
  PNGs up to 960 x 960): https://onrobot.com/en/products/rg2-finger-gripper
  - e.g. https://onrobot.com/img/paths/storage/products/rg2_newqc.png/fa69d3029d1baf888760a3942f6a344d/rg2_newqc.png
- Copyright: OnRobot A/S.

## d405.png — Intel RealSense Depth Camera D405

- File: `d405.png`, 1920 x 667 px, PNG (palette), white background, 185 KB.
  Shows the D405 next to a golf ball (the manufacturer's own scale reference;
  the camera is 42 x 42 x 23 mm). Crop the wide white margins if needed.
- Direct image URL:
  https://www.realsenseai.com/wp-content/uploads/2025/07/D-405.png
- Source page: https://www.realsenseai.com/products/stereo-depth-camera-d405/
  (RealSense was spun out of Intel in 2025; the old
  https://www.intelrealsense.com/depth-camera-d405/ no longer resolves.)
- Other official product shots on the same page, if a different angle is wanted:
  - https://www.realsenseai.com/wp-content/uploads/2025/06/405-camera-2.jpg (1340 x 740, rear with Intel RealSense branding)
  - https://www.realsenseai.com/wp-content/uploads/2025/06/disassembled-1.png (1446 x 904, exploded view)
- Copyright: RealSense (formerly Intel RealSense).

## Local search (no suitable product photos found)

Searched /home/anton/projects/robotics, ~/Downloads, ~/Pictures and
~/Documents/vbti for `*ur5*`, `*rg2*`, `*d405*`, `*realsense*`, `*onrobot*`.
Only hits were D405 scene captures, a hand-eye diagram and 256 px USD
thumbnails — none usable as product images.


Note: hardware.png in the parent folder is the montage of ur5e_datasheet_cutout.png, rg2.jpg and a SAM 3 cut-out of d405.png (d405_cam.png).
