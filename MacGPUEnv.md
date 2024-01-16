# How to use GPU on Mac M1/2

1. Install xcode tools

```
xcode-select --install
```

2. Install llvm

```
brew install llvm libomp
```

3. 'requirements_for_mac.txt' instead of 'requirements.txt'

```
pip install -r requirements_for_mac.txt
```