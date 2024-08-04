AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
docker build -t water-probability-estimator-repo .
docker tag water-probability-estimator-repo:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/water-probability-estimator-repo:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/water-probability-estimator-repo:latest
