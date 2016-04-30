var resevoirSampleInd = function(arr, numPts){
    let newArr = [];
    for(let i = 0; i < numPts; i++){
        newArr.push(i);
    }

    for(let i = numPts; i < arr.length; i++){
        let j = Math.floor((Math.random() * i));
        if(j < numPts){
            newArr[j] = i;
        }
    }

    return newArr;
}

var getArrayFromInds = function(arr, inds){
    let newArr = [];
    for(let i = 0; i < inds.length; i++){
        newArr.push(arr[inds[i]]);
    }
    return newArr;
}

module.exports = {resevoirSampleInd, getArrayFromInds};
