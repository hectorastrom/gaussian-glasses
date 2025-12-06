# Downloading Dataset from AWS:
1. Make sure you're in diffusers-exploration/
1. Download: `aws s3 cp s3://hectorastrom-dl-final/datasets/COD10K-v3.zip ./data/COD.zip`
1. Unzip: `unzip ./data/COD.zip -d ./data/`
1. Rename: `mv data/COD10K-v3 data/COD10K`
1. Clean up: `rm -rf data/COD.zip`