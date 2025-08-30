# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AWS CDK (Cloud Development Kit) TypeScript project that deploys a Grafana JSON API data source using AWS Lambda. The project is called "DuckFactory" and simulates factory production metrics.

## Common Development Commands

### Build and Development
- `npm run build` - Compile TypeScript to JavaScript
- `npm run watch` - Watch for changes and compile automatically
- `npm run test` - Run Jest unit tests
- `npm run deploy` - Deploy stack to AWS without approval prompt

### CDK Commands
- `npx cdk deploy` - Deploy the stack to AWS
- `npx cdk diff` - Compare deployed stack with current state
- `npx cdk synth` - Synthesize CloudFormation template
- `npx cdk destroy` - Remove the stack from AWS

### Testing
- `npm test` - Run all tests
- `npm test -- --watch` - Run tests in watch mode
- `npm test -- <test-file-name>` - Run a specific test file

## Architecture Overview

### Stack Structure
- **Main Stack**: `lib/cdk-stack.ts` - Defines DuckFactoryStack with Lambda function and Function URL
- **Lambda Function**: `lambda/json-api/index.ts` - Implements Grafana JSON API endpoints
- **Entry Point**: `bin/cdk.ts` - CDK app initialization

### Lambda Function Details
The Lambda function (`DuckFactory_Json_API`) implements a Grafana JSON API data source with:
- **Endpoints**: `/metrics` (returns available metrics) and `/query` (returns metric data)
- **Metrics Available**:
  - Temperature (celsius)
  - Humidity (percentage)
  - Production count
  - Atmospheric pressure
  - Daily and monthly aggregated data
- **CORS**: Enabled with permissive settings for development
- **Access**: Public via Function URL with no authentication

### Key Technical Details
- TypeScript with ES2022 target and strict mode enabled
- Node.js 18 runtime for Lambda
- Jest for testing with TypeScript support
- CDK v2 with all feature flags enabled for v2 behavior