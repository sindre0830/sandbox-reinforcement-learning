# Python Template

A template repository for Python projects.

## Setup

0. Install Python 3.12 from [python.org](https://www.python.org/downloads/)
1. Install Poetry from [python-poetry.org](https://python-poetry.org/docs/)
2. Install project dependencies:
    ```sh
    poetry install
    ```
3. Run the project:
    ```sh
    poetry run main
    ```
4. Run tests:
    ```sh
    poetry run pytest
    ```

### Development

To add packages, run:

- For regular dependencies:
    ```sh
    poetry add PACKAGE_NAME --lock
    ```
- For development dependencies:
    ```sh
    poetry add PACKAGE_NAME --group dev --lock
    ```

After that, follow the setup guide from step 2
