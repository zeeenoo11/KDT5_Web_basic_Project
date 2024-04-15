function changePage(pageName) {
  var iframe = document.getElementById("contentFrame");
  iframe.src = pageName;
}
function changeImage() {
  const pikachu = document.getElementById("pikachu");
  pikachu.src = "pikachu2.png"; // 마우스를 올렸을 때 바뀔 이미지의 경로
}

function restoreImage() {
  const pikachu = document.getElementById("pikachu");
  pikachu.src = "pikachu.png"; // 마우스를 내렸을 때 다시 복원될 이미지의 경로
}
