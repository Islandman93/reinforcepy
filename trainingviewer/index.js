var d3 = require('d3');
var c3 = require('c3');
var sampling = require('./sampling');
var $ = require('jquery');

var chart = generateChart();
var masterData = [];
var maxPoints = 0;
let dataSeries = {};

// start data get
$.get("http://localhost:8080/stats/getlist", function(data){
    dataList = JSON.parse(data);
    maxPoints = Math.round(1000/dataList.length);

    // iterate over data and add to chart
    for(let i = 0; i < dataList.length; i++){
        d3.json("http://localhost:8080/stats/getstat/"+dataList[i], function(error, json){
            if (error) return console.warn(error);
            let columnData = parseData(json, dataList[i]);
            masterData.push({x: columnData[0], y: columnData[1]});
            let columnXs = {};
            columnXs[dataList[i]] = 'x'+dataList[i];
            chart.load({
                xs: columnXs,
                columns: columnData
            });
        });
    }
});

function parseData(data, key){
    // get x y arrays
    let values = data.filter((x) => x.type === 'Minibatch');
    let sampleInd = sampling.resevoirSampleInd(values, maxPoints);
    let sampledValues = sampling.getArrayFromInds(values, sampleInd);
    let y = sampledValues.map((y) => y.loss);
    let x = sampledValues.map((x) => x.step);
    // unshift to put the name infront
    y.unshift(key);
    x.unshift('x'+key);

    return [x, y];
}

function generateChart(){
    var chart = c3.generate({
            bindto: '#vis',
            data: {
                columns: [],
                type: 'scatter'
            },
            zoom: {
                enabled: true,
                rescale: true
            },
        });
    return chart
}
