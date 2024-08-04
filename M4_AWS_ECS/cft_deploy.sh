aws cloudformation deploy  \
    --stack-name water-probability-estimator-stack \
    --no-fail-on-empty-changeset \
    --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
    --region us-east-1 \
    --template-file "M4_AWS_ECS/aws_cft.yml"