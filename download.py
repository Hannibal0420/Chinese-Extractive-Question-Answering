import gdown

url = 'https://drive.google.com/drive/folders/1CTx4A8GucqFoSdOyHCFKrcC_X9sZ_3-P?usp=share_link'

gdown.download_folder(url, quiet=True, use_cookies=False)