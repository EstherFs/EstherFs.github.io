var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
  };
  reader.readAsDataURL(input.files[0]);
}

function transfer() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("transfer").disabled = true;
  el("transfer").innerHTML = "Transfering...";

  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      const commodity_image = el("transfer-result");
      var response = JSON.parse(e.target.responseText);

      let image_r = response["result"];
      let imageSRC = "data:image/jpeg;base64," + image_r;
      commodity_image.src = imageSRC
      
    }
    el("transfer").disabled = false;
    el("transfer").innerHTML = "Transfer";

  };
  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}

new fullpage('#fullpage', {
  //options here
  autoScrolling:true,
  scrollVertically: true,
  loopTop: true,
  loopBottom: true,
  paddingTop: String(document.getElementById("navbar").clientHeight)+'px',
  anchors:['story_part', 'painting_part', 'painter_part', 'transfer_part'],
  licenseKey: "gplv3-license"
});