// Triggering function of py by passing Querry through JS

function Initiallization() {
    var query = document.getElementById("data_1").value;
    eel.Initiallization(query)(callback);
  }
  function callback(Retrieved) {
    document.getElementById("length").value = Retrieved.length;
    if(Retrieved.length==0){
      document.getElementById("length").value = 0;
      document.getElementById("Retrieved").value = 0;

    }
    document.getElementById("Retrieved").value = Retrieved;
  }