from imutils import paths
from PIL import Image


def stitching():
  # Change file path for the image_folder
  image_folder = f'C:\\Users\\grego\\OneDrive\\Documents\\MDP CZ3004\\yolov5-master\\yolov5-master\\runs\\detect'
  imagePaths = list(paths.list_images(image_folder))
  images = [Image.open(x) for x in imagePaths]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('runs/stitched/stitchedOutput.jpg', format='jpg')