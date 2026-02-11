# Testing Guide

FAI uses `xunit.v3` with Microsoft Testing Platform (MTP) for testing.

## Running Tests

All tests are located in the `test/` directory and use `net10.0`.

### Command Line

To run all tests:

```bash
dotnet test
```

To run tests for a specific project:

```bash
dotnet test test/FAI.Core.Tests/FAI.Core.Tests.csproj
```

### Visual Studio Code

You can use the built-in Test Explorer in VS Code to run and debug tests.

## Infrastructure

The testing infrastructure is centralized in `test/Directory.Build.props`, which includes:
- `xunit.v3`
- `Microsoft.NET.Test.Sdk`
- MTP runner enabled via `<UseMicrosoftTestingPlatformRunner>true</UseMicrosoftTestingPlatformRunner>`

## Project Coverage

- `FAI.Core.Tests`
- `FAI.Core.Extensions.DI.Tests`
- `FAI.NLP.Tests`
- `FAI.NLP.Extensions.DI.Tests`
- `FAI.Onnx.Tests`
- `FAI.Vision.Tests`
- `FAI.Extensions.Evaluation.Tests`
