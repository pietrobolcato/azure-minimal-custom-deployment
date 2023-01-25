echo "- Creating model..."
az ml model create -f ../model.yaml
echo "- Model created succesfully"

echo "- Creating endpoint..."
az ml online-endpoint create -f endpoint.yaml
echo "- Endpoint created succesfully"