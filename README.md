# ChainSniper: A Machine Learning Approach for Auditing Cross-Chain Smart Contracts

This repository presents ChainSniper, a sidechain-based framework that integrates machine learning for automated vulnerability assessment of cross-chain smart contracts. The system aims to enhance security across interconnected blockchain networks by leveraging advanced techniques to identify and mitigate potential vulnerabilities in smart contracts operating across multiple chains.

## Overview

As blockchain technology continues to evolve, the need for interoperability among diverse networks has become increasingly crucial. However, the presence of vulnerabilities in cross-chain smart contracts poses significant security risks, potentially leading to financial losses and compromised systems. ChainSniper addresses this challenge by providing an automated and scalable approach to auditing smart contracts across interconnected blockchain networks.

## Features

- **Sidechain Integration**: ChainSniper utilizes a sidechain architecture to facilitate secure inter-blockchain data exchange, enabling seamless communication and transfer of smart contract information between heterogeneous networks.

- **Machine Learning Models**: The system incorporates various machine learning and deep learning models, including Decision Trees, Random Forests, XGBoost, CNN, LSTM, SVMs with different kernels, Logistic Regression, Feed Forward Neural Networks, and RoBERTa. These models are trained on a novel dataset, "CrossChainSentinel," to accurately detect vulnerabilities in cross-chain smart contracts.

- **CrossChainSentinel Dataset**: A comprehensive dataset comprising 300 manually labeled smart contract samples, including 158 benign contracts and 142 contracts with injected vulnerabilities such as reentrancy flaws, overflow and underflow bugs, and unprotected ether withdrawal issues.

- **Vulnerability Detection**: ChainSniper is capable of identifying and flagging various vulnerabilities commonly found in cross-chain smart contracts, including reentrancy attacks, integer overflow/underflow issues, and unprotected ether withdrawal vulnerabilities.

## Experiments and Results

The performance of ChainSniper has been thoroughly evaluated through a series of experiments. The system achieved remarkable results, with the RoBERTa model demonstrating the highest accuracy of 0.967, precision of 0.999, recall of 0.875, and an F1 score of 0.933 in detecting cross-chain smart contract vulnerabilities.

## Contributing

Contributions to this project are welcome. Please follow the standard guidelines for contributing to open-source projects.

## Acknowledgments

This research is funded by the Faculty of Computer Networks and Communications, University of Information Technology, Vietnam National University Ho Chi Minh City, Vietnam.

## References

The research paper describing the ChainSniper system and its underlying concepts can be found at:

Tran, T.-D., Vo, K.A., Duy, P.T., Cam, N.T., & Pham, V.-H. (2024). ChainSniper: A Machine Learning Approach for Auditing Cross-Chain Smart Contracts. In Proceedings of ICIIT 2024 International Conference on Intelligent Information Technology (ICIIT2024). 