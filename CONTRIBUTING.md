# 🤝 Contributing to `Modalbed`

We welcome contributions from the community to help improve and expand the project. Below are some guidelines to help you get started.

## How to Contribute

### Reporting Issues

- If you encounter any bugs or have suggestions for improvements, please open an issue.
- Provide as much detail as possible, including steps to reproduce the issue and any relevant logs or screenshots.

### Pull Requests

- Fork the repository and create your branch from `main`.
- Ensure your code follows the project's coding standards and passes all tests.
- Write clear, concise commit messages and include a detailed description of your changes in the pull request.
- If your pull request addresses an issue, please reference it in the description.

### Adding New Features

- **Datasets**: 
  - Add a download function in `modalbed/download.py`.
  - Add the dataset class for loading in `modalbed/datasets.py`.

- **Perceptors**:
  - Add a download function in `modalbed/download.py`.
  - Add the perceptor class in `modalbed/perceptors/<CUSTOM PERCEPTOR>.py`.

- **Algorithms**:
  - Add your algorithm in the `modalbed/algorithms` folder.
  - Update `modalbed/algorithms/__init__.py` to register your algorithm.

We appreciate your contributions and look forward to working together to make `Modalbed` even better!

Welcome your pull request 😊!