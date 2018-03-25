
function showSmartReply(){
      document.getElementById("getResponseBtn").collapsed = 0;
      document.getElementById("buttonGrp").collapsed = 1;
      window.removeEventListener("click", showSmartReply);
}
setTimeout(function(){
  document.getElementById("getResponseBtn").addEventListener("click", function(){getResponses()});
}, 1000);
function getResponses(){
  setTimeout(function(){window.addEventListener("click", showSmartReply);}, 400);
  document.getElementById("getResponseBtn").collapsed = true;
  var xmlhttp = new XMLHttpRequest();
  var url = "http://localhost:8888/";
  xmlhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
      var responses = JSON.parse(this.responseText)['responses'];
      for(var i = 0; i < 3; i++){
        document.getElementById("buttonGrp").childNodes[i].innerHTML = responses[i];
        document.getElementById("buttonGrp").childNodes[i].addEventListener("click", function(e){newMsg(this);});
        document.getElementById("buttonGrp").collapsed = false;
      }
    }
  };
  xmlhttp.open("GET", url, true);
  xmlhttp.send();
}

function newMsg(el){
  var mail_id = gFolderDisplay.selectedMessage.author.match(/<(.*?)>/);
  mail_id = mail_id[mail_id.length - 1]
  var subj = "Re:" + gFolderDisplay.selectedMessage.subject
  var sURL="mailto:"+mail_id+"?subject="+subj+"&body="+el.innerHTML;

  var msgComposeService=
    Components.classes["@mozilla.org/messengercompose;1"]
    .getService(Components.interfaces.nsIMsgComposeService);

  // make the URI
  var ioService =
    Components.classes["@mozilla.org/network/io-service;1"]
      .getService(Components.interfaces.nsIIOService);

  aURI = ioService.newURI(sURL, null, null);

  // open new message
  msgComposeService.OpenComposeWindowWithURI (null, aURI);

}
// window.setInterval(
//     function() {
//         findAccountFromFolder();
// //     }, 60000); //update date every minute
// Cu.import("resource:///modules/iteratorUtils.jsm");
// // function listMessages(aFolder) {
// //   let database = aFolder.msgDatabase;
// //   for each (let msgHdr in fixIterator(database.EnumerateMessages(), Ci.nsIMsgDBHdr)) {
// //     let title = msgHdr.mime2DecodedSubject;
// //     console.log(title+"\n");
// //     // do stuff with msgHdr
// //   }
// //   // don't forget to close the database
// //   aFolder.msgDatabase = null;
// // }
// function findAccountFromFolder (theFolder) {
//     if (!theFolder)
//         return null;
//     var acctMgr = Components.classes["@mozilla.org/messenger/account-manager;1"]
//         .getService(Components.interfaces.nsIMsgAccountManager);
//     var accounts = acctMgr.accounts;
//     for (var i = 0; i < accounts.Count(); i++) {
//         var account = accounts.QueryElementAt(i, Components.interfaces.nsIMsgAccount);
//         var rootFolder = account.incomingServer.rootFolder; // nsIMsgFolder
//         if (rootFolder.hasSubFolders) {
//             var subFolders = rootFolder.subFolders; // nsIMsgFolder
//             while(subFolders.hasMoreElements()) {
//                 if (theFolder == subFolders.getNext().QueryInterface(Components.interfaces.nsIMsgFolder))
//                     console.log(account.QueryInterface(Components.interfaces.nsIMsgAccount));
//             }
//         }
//     }
//     return null;
// }
// // /*function startup() {
//     var myPanel = document.getElementById("my-panel");
//     var date = new Date();
//     var day = date.getDay();
//     var dateString = date.getFullYear() + "." + (date.getMonth()+1) + "." + date.getDate();
//     myPanel.label = "Date: " + dateString;
// }*/
