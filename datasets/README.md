# Downloading Dataset from AWS:
1. Make sure you're in diffusers-exploration/
1. Download: `aws s3 cp s3://hectorastrom-dl-final/datasets/COD10K-v3.zip ./datasets/COD.zip`
1. Unzip: `unzip ./datasets/COD.zip -d ./datasets/`
1. Rename: `mv datasets/COD10K-v3 datasets/COD10K`
1. Clean up: `rm -rf datasets/COD.zip`