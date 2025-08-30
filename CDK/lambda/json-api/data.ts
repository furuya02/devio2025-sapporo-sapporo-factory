import { start } from "repl";
import { ParseRequest } from "./parseRequest";
import { randomInt } from "crypto";

const OEE = "oee";
const PRODUCT_TEMPERATURE = "production_count"; // 生産数
const PRODUCT_RATE = "production_rate"; // 生産率

// const TEMPERATURE = "temperature_celsius";
// const HUMIDITY = "humidity_percentage";
// const PRESSURE = "atmospheric_pressure_hpa";
// const MONTHLY = "monthly_data";
// const DAILY = "daily_data";
const METRICS = [
  OEE,
  PRODUCT_TEMPERATURE,
  PRODUCT_RATE,
  // TEMPERATURE,
  // HUMIDITY,
  // PRESSURE,
  // MONTHLY,
  // DAILY,
];

export function metrics(): any[] {
  const response = [];
  for (const metric of METRICS) {
    response.push({
      text: metric.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      value: metric,
    });
  }
  return response;
}

export function query(req: ParseRequest): any[] {
  const response = [];
  const startTime = req.requestData["startTime"];
  const maxDataPoints = req.requestData["maxDataPoints"];

  for (const target of req.requestData.targets) {
    const target_name = target["target"];
    const maxDataPoints = req.requestData.maxDataPoints;
    const payload = target["payload"];

    if (target_name === "oee") {
      const part = Number(payload.part);
      const index = part - 1;
      const min = new Date(startTime).getMinutes();
      const col = Math.floor(min / 3);

      const table = [
        [
          87.5, 87.0, 87.5, 87.7, 87.8, 87.9, 87.5, 87.0, 87.5, 87.7, 87.5,
          87.0, 87.5, 87.7, 87.8, 87.9, 87.5, 87.0, 87.5, 87.7,
        ],
        [
          71.4, 72.0, 73.5, 72.5, 73.2, 74.1, 75.0, 74.5, 74.0, 73.5, 73.0,
          72.5, 72.0, 71.5, 71.0, 70.5, 70.0, 69.5, 69.0, 68.5,
        ],
        [
          93.2, 93.0, 93.2, 94.0, 93.8, 98.7, 97.9, 97.5, 97.0, 97.5, 97.7,
          97.5, 97.0, 97.5, 97.7, 97.8, 97.9, 97.5, 97.0, 97.5, 97.7,
        ],
      ];
      let val =
        index === 3
          ? (table[0][col] / 100) *
            (table[1][col] / 100) *
            (table[2][col] / 100) *
            100
          : table[index][col];

      const timestamp = new Date().getTime();
      response.push({
        target: `part:${part}`,
        datapoints: [[val, timestamp]],
      });
      // } else if (target_name === "humidity_percentage") {
      //   response.push({
      //     target: target_name,
      //     datapoints: [
      //       [65, new Date("2025-08-17T08:11:42Z").getTime()],
      //       [68, new Date("2025-08-17T08:12:42Z").getTime()],
      //       [70, new Date("2025-08-17T08:13:42Z").getTime()],
      //       [72, new Date("2025-08-17T08:20:42Z").getTime()],
      //       [75, new Date("2025-08-17T08:21:42Z").getTime()],
      //     ],
      //   });
    } else if (target_name === "production_rate") {
      // 生産率
      const line = Number(payload.line);
      const val = 90 + randomInt(0, 100) / 10;
      const timestamp = new Date().getTime();
      response.push({
        target: `line:${line}`,
        datapoints: [[val, timestamp]],
      });
    } else if (target_name === "production_count") {
      // 生産数
      const hours = 6 + 1;
      const line = Number(payload.line);
      const table = [
        [90, 90, 90, 90, 90, 90, 90],
        [85, 88, 87, 80, 82, 88, 80],
        [90, 95, 96, 99, 92, 91, 91],
        [81, 82, 89, 81, 81, 88, 90],
        [91, 91, 91, 91, 92, 93, 92],
      ];
      for (let hours = 0; hours <= 6; hours++) {
        table[0][hours] += randomInt(0, 10) / 10;
        table[1][hours] += randomInt(0, 10) / 10;
        table[2][hours] += randomInt(0, 10) / 10;
        table[3][hours] += randomInt(0, 10) / 10;
        table[4][hours] += randomInt(0, 10) / 10;
      }
      response.push({
        target: `line:${line}`,
        datapoints: createDataPoints(startTime, hours, line, table),
      });
    }
  }
  return response;
}
function createDataPoints(
  startTime: number,
  hours: number,
  line: number,
  table: number[][]
) {
  const st = new Date(startTime);
  st.setHours(st.getHours() - hours);
  const dataPoints = [];
  for (let i = 0; i < hours; i++) {
    st.setHours(st.getHours() + 1);
    dataPoints.push([table[line][i], st.getTime()]);
  }
  return dataPoints;
}

// function createTimestamp(jstTimeStr: string): number {
//   return new Date(`${jstTimeStr}+09:00`).getTime();
// }
