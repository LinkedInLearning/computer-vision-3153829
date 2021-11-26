# Add a repository to the apt list
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Add a GPG key to the apt sources
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update apt packages
sudo apt-get update

# Install TensorFlow Lite
sudo apt-get install python3-tflite-runtime

