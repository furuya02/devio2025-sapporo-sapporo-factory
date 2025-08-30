import { APIGatewayProxyResult, Context } from "aws-lambda";
import { LambdaEvent, ParseRequest } from "./parseRequest";
import { createOptionResponse, createResponse } from "./util";
import { metrics, query } from "./data";

export const handler = async (
  event: LambdaEvent,
  context: Context
): Promise<APIGatewayProxyResult> => {
  console.log("🐹 event:", JSON.stringify(event, null, 2));

  const req = new ParseRequest(event, context);
  if (req.failed) {
    return createResponse(400, { error: "Bad Request" });
  }

  let responseBody: any[] = [];
  switch (req.httpMethod) {
    case "GET":
      return createResponse(200, {
        message: "",
      });
    case "OPTIONS":
      return createOptionResponse();
    case "POST":
      if (req.path === "/metrics") {
        responseBody = metrics();
      } else if (req.path === "/query") {
        responseBody = query(req);
      } else {
        responseBody = [{ error: "Not Found" }];
      }
  }
  const responseBodyStr = JSON.stringify(responseBody, null, 2);
  console.log(`🐹 response:${responseBodyStr}`);

  return {
    statusCode: 200,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    },
    body: responseBodyStr,
  };
};
