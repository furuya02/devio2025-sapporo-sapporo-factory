import {
  APIGatewayProxyEvent,
  APIGatewayProxyEventQueryStringParameters,
  Context,
} from "aws-lambda";

export type LambdaEvent = APIGatewayProxyEvent | FunctionUrlEvent;

interface FunctionUrlEvent {
  version: string;
  requestContext: {
    http: {
      method: string;
    };
    requestId: string;
  };
  rawPath: string;
  queryStringParameters?: Record<string, string> | null;
  headers: Record<string, string>;
  body?: string;
  isBase64Encoded: boolean;
}

function isFunctionUrlEvent(event: LambdaEvent): event is FunctionUrlEvent {
  return (
    "rawPath" in event &&
    "requestContext" in event &&
    "http" in (event as FunctionUrlEvent).requestContext
  );
}

export class ParseRequest {
  httpMethod: string;
  path: string;
  queryParams: APIGatewayProxyEventQueryStringParameters;
  //body: string;
  headers: any;
  timestamp: any;
  requestData: any;
  failed: boolean = false;

  constructor(event: LambdaEvent, context: Context) {
    this.timestamp = context.awsRequestId;

    this.httpMethod = isFunctionUrlEvent(event)
      ? event.requestContext.http.method
      : (event as APIGatewayProxyEvent).httpMethod || "GET";
    this.path = isFunctionUrlEvent(event)
      ? event.rawPath
      : (event as APIGatewayProxyEvent).path || "/";
    this.queryParams = event.queryStringParameters || {};

    // contentType = (
    //   this.headers["content-type"] ||
    //   this.headers["Content-Type"] ||
    //   ""
    // ).toLowerCase();
    this.headers = event.headers;

    let body = event.body || "";
    if (body && event.isBase64Encoded) {
      body = Buffer.from(body, "base64").toString("utf-8");
    }

    if (body && this.headers["content-type"]?.includes("application/json")) {
      try {
        this.requestData = JSON.parse(body);
      } catch (error) {
        console.log(error);
        this.failed = true;
      }
    }

    console.log(`httpMethod:${this.httpMethod}`);
    console.log(`path:${this.path}`);
    console.log(`headers:${JSON.stringify(this.headers)}`);
    console.log(`queryParams:${JSON.stringify(this.queryParams)}`);
    console.log(`requestData:${JSON.stringify(this.requestData)}`);
  }
}
