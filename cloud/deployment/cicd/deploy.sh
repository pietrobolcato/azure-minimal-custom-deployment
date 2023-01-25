ENDPOINT="mnist-endpoint-865021"

echo "- Creating model..."
az ml model create -f ../model.yaml --tags "release-id=$GITHUB_REF_NAME" "author=$GITHUB_ACTOR"
echo "- Model created succesfully"

echo "- Checking if endpoint exists already"

if [[ $(az ml online-endpoint list --query "[?name=='$ENDPOINT'] | length(@)") > 0 ]]
then
    echo "- Endpoint exists"
else
    echo "- Endpoint doesn't exist"
    echo "- Creating endpoint..."

    az ml online-endpoint create -f ../endpoint.yaml
    
    echo "- Endpoint created succesfully"
fi

