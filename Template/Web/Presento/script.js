const img= document.getElementById('imagecheque');

const predictionletter= document.getElementById('predictionletter');
const predictionnumber= document.getElementById('predictionnumber');

const file= document.getElementById('fileupload');
myfunction = async()=>{var path = (window.URL || window.webkitURL).createObjectURL(file.files[0]);
   img.src=path;
   
  var formData = new FormData();
  formData.append("img", file.files[0]);


const dataToSend = formData;
let dataReceived = ""; 
const res= await fetch("http://3.93.59.211:8000/test", {
    method: "POST",
   
    //headers: { "Content-Type": "multipart/form-data" },
    body: dataToSend
})
var data = await res.json();
console.log(data)
console.log(predictionletter);
predictionletter.innerHTML=data.letter;
predictionnumber.innerHTML=data.number;
   


console.log(`Received: ${dataReceived}`) 

}
file.addEventListener('change',function async (){
	//var tmppath = URL.createObjectURL(file.mozFullPath);
   // $("img").fadeIn("fast").attr('src',URL.createObjectURL(event.target.files[0]));
  myfunction()
   




})
