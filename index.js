const fs = require("fs");
const Papa = require("papaparse");
const map = require("lodash/map");
const forEach = require("lodash/forEach");
global.XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest;
console.log("starting");

// npm install http-server -g
// cd to your directory where you want to server files, then run `http-server`

// file url
const url = "http://127.0.0.1:8080/wrangleeo.csv";
const outputFile = "papa.csv";

const complete = (results, file) => {
  // This is mapping each row (value) to a new value
  const newResult = map(results.data, (value, key) => {
    // value is each row as an array of strings
    if (value[1] !== "genres") {
      const genreString = value[1].replace(/'/g, '"');
      const genres = JSON.parse(genreString);
      while (genres.length < 4) {
        genres.push("");
      }
      forEach(genres, (obj, index) => {
        value[index + 1] = obj.id;
      });
    }
    return value;
  });
  const csv = Papa.unparse(newResult);
  console.log("DONE — writing to file");
  fs.writeFile(outputFile, csv, err => {
    console.log("done?");
  });
};

Papa.parse(url, {
  download: true,
  complete
});
