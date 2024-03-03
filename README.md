# Federated Learning for Web3 Research

## Overview
This repository contains code for implementing federated learning on the CIFAR-10 dataset as part of personal research to develop a federated learning system for Web3. The project, titled Cortex MVP, aims to explore the application of federated learning in decentralized environments leveraging Web3 technologies. Federated learning, a decentralized machine learning approach, distributes the model training process across multiple nodes without centralizing the data. The research investigates the feasibility and effectiveness of utilizing federated learning within Web3 ecosystems.

## Contents
- `client.py`: Defines the client for federated learning.
- `server.py`: Starts the server to coordinate the federated learning process.
- `README.md`: The document you are currently reading, providing an overview of the project.

## Usage
1. **Setup Environment**: Ensure Python and necessary dependencies are installed. You can install them using `pip install -r requirements.txt`.

2. **Starting Federated Learning Server**: Run the `server.py` script to start the federated learning server.
    ```bash
    python server.py
    ```

3. **Federated Learning Clients**: Clients can connect to the server for federated learning. Modify and utilize the `client.py` script according to your setup and requirements.

## Customization
You can customize the client behavior and server configuration as per your requirements by modifying respective files. Flower was chosen as the initial framework due to its simplicity, but it can potentially be replaced in the future as the project develops.

## Credits
This project utilizes the federated learning framework. Credits to the respective development teams for providing efficient and scalable platforms for federated learning.

## License
This project is licensed under the MIT License.

Feel free to extend and modify this project according to your needs. Contributions are welcome!
