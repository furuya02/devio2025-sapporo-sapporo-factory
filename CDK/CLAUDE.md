# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the AWS CDK infrastructure for the DuckFactory IoT system, which provides cloud services for a duck factory quality control system. The CDK stack deploys multiple AWS services including Lambda functions, DynamoDB, S3, and IAM users to support both the Grafana monitoring interface and IoT edge device connectivity.

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
- `npx cdk ls` - List all stacks in the app

### Testing
- `npm test` - Run all tests
- `npm test -- --watch` - Run tests in watch mode
- `npm test -- <test-file-name>` - Run a specific test file

### TypeScript Validation
- `npx tsc --noEmit` - Type check without building

## Architecture Overview

### Stack Components (`lib/cdk-stack.ts`)

The main CDK stack deploys five key infrastructure components:

1. **JSON API Lambda** (`JsonApiConstruct`):
   - Grafana-compatible JSON API endpoints
   - Function URL with public CORS-enabled access
   - Node.js 20.x runtime with 30s timeout

2. **DynamoDB Table** (`DynamoDBConstruct`):
   - Time-series optimized table for IoT data
   - Partition key: `deviceId`, Sort key: `timestamp`
   - Three GSIs for efficient querying:
     - `LineNameTimeIndex`: Query by production line
     - `StatusTimeIndex`: Query by duck status (pass/fail)
     - `DateTimeIndex`: Query by date partition
   - TTL enabled on `ttl` attribute for automatic data cleanup
   - Pay-per-request billing mode

3. **S3 Bucket** (`S3Construct`):
   - Storage for duck images from quality control
   - 90-day lifecycle rule for automatic cleanup
   - Bucket name: `duck-factory-229914322323`

4. **IAM User** (`UserConstruct`):
   - Service account for Jetson edge devices
   - Permissions:
     - Bedrock: Access to Titan embedding model
     - S3: Read/write to duck factory bucket
     - IoT Core: Publish to `duck-factory/*` topics and device shadows
     - DynamoDB: Full CRUD operations on factory table

5. **IoT Rule** (`IoTRuleConstruct`):
   - Routes MQTT messages from `duck-factory/+` topics to DynamoDB
   - SQL transformation adds metadata:
     - `lineName`: Extracted from topic (e.g., "line-01" from "duck-factory/line-01")
     - `timestamp`: Message timestamp
     - `datePartition`: Date in YYYY-MM-DD format for efficient queries
   - Uses DynamoDBv2 action for flexible message storage

### Directory Structure
```
CDK/
├── bin/cdk.ts              # CDK app entry point
├── lib/
│   ├── cdk-stack.ts        # Main stack definition
│   └── construct/          # Reusable CDK constructs
│       ├── dynamodb.ts     # DynamoDB table construct
│       ├── iot-rule.ts     # IoT Rule construct
│       ├── jetson-api.ts   # Lambda function construct
│       ├── s3.ts           # S3 bucket construct
│       └── user.ts         # IAM user construct
├── lambda/
│   └── json-api/           # Lambda function code
│       ├── index.ts        # Main handler
│       ├── data.ts         # Metric data generation
│       ├── parseRequest.ts # Request parsing utilities
│       └── util.ts         # Helper functions
└── test/                   # Jest unit tests
```

### Key Technical Details

- **TypeScript Configuration**:
  - Target: ES2022
  - Module: NodeNext
  - Strict mode enabled
  - Source maps inline for debugging

- **Lambda Runtime**: Node.js 20.x (upgraded from 18.x)
- **CDK Version**: 2.x with all v2 feature flags enabled
- **Testing**: Jest with ts-jest transformer

### Infrastructure Outputs

The stack exports the following CloudFormation outputs:
- `FunctionUrlOutput`: JSON API endpoint URL
- `DuckFactoryUserAccessKeyId`: IAM access key ID
- `DuckFactoryUserSecretAccessKey`: IAM secret access key
- `DuckFactoryTableName`: DynamoDB table name
- `DuckFactoryTableArn`: DynamoDB table ARN
- `IoTRuleName`: IoT Rule name for MQTT to DynamoDB routing

## IoT Permissions

The IAM user has restricted IoT Core permissions:
- Connect with client ID matching Thing name
- Publish to `duck-factory/*` topics only
- Access device shadow topics for Things

## Development Notes

- All resources use `RemovalPolicy.DESTROY` for easy cleanup in development
- CORS is permissively configured for development (all origins allowed)
- Lambda function uses Function URLs instead of API Gateway for simplicity
- DynamoDB uses on-demand billing to avoid provisioning concerns