var tmppath;
var doc_name;
var doc_type = [];
var addpersonalElm = `<div class="added-elem">
                  <select>
                    <option value="-1">--Select--</option>
                    <option value="pdf">Pdf File</option>
                    <option value="text">Text File</option>
                </select>
                <input type="text" readonly placeholder="File Name"/>
                <div class="upload-btn-wrapper">
                    <span class="folder-img">&nbsp;</span>
                    <input type="file" name="myfile" />
                </div>
                <a class="thumb">

                </a>
            </div>`;
var addEduElm = `<div class="added-elem">
                  <select>
                    <option value="-1">--Select--</option>
                    <option value="tenth">10th</option>
                    <option value="twelve">12th</option>
                    <option value="grad">Graduation</option>
                    <option value="pg">Post Graduation</option>
                </select>
                <input type="text" readonly placeholder="File Name"/>
                <div class="upload-btn-wrapper">
                    <span class="folder-img">&nbsp;</span>
                    <input type="file" name="myfile" />
                </div>
                <a class="thumb">

            </div>`;

var addProfElm = `<div class="added-elem">
                <input type="text" readonly placeholder="File Name"/>
                <div class="upload-btn-wrapper">
                    <span class="folder-img">&nbsp;</span>
                    <input type="file" name="myfile"/>
                </div>
                <a class="thumb">

            </div>`;

function submit() {
    var allImages = document.querySelectorAll("a.thumb");
    var allElemProf = $(".professional-info .added-elem");
    var allElem = $(".personal-info .added-elem");
    var upload_data = [];
    $(allElem).map(function(index, thisinst) {
        $(thisinst).find("select").val() !== -1 && $(thisinst).find("select").val() !== "-1" && $(thisinst).find("select").val() !== undefined ? upload_data.push({ "doc_type": $(thisinst).find("select").val(), "doc_name": $(thisinst).find("input[type=text]").val() }) : "";
    });
    $(allElemProf).map(function(index, thisinst) {
        if ($(thisinst).find("input[type=text]").val() !== "") {
            var indexMap = parseInt(index + 1);
            $(thisinst).find("a.thumb").attr("doc_type", "exp" + indexMap);
            upload_data.push({ "doc_type": "exp" + indexMap, "doc_name": $(thisinst).find("input[type=text]").val() });
        }
    });
    //if (allElem.length !== upload_data.length) {
    //     alert("Please upload document before.");
    //     return
    // }
    //      "doc_name": doc_name,
    //    "doc_type": $("#docType").val()
    $(".loader").show();
    for (var i = 0; i < allImages.length; i++) {
        allImages[i].click();
    }
    // setTimeout(function() {
        var serviseBaseUrl = "http://127.0.0.1:5000";
        $.support.cors = true;
       $.ajax({
               type: "GET",
               url:"http://127.0.0.1:5000/textSum",
               data: JSON.stringify(upload_data),
               dataType: "jsonp",
               // contentType: "application/json; charset=utf-8",

       error: function (xhr, ajaxOptions, thrownError) {

           alert("sucesse");
           alert("Summary has been saved in local downloads");
           $(".loader").hide();

      },
       success:function(result){
             alert("sucesse");
             alert("Summary has been saved in local downloads");
             // console.log(result);
             $(".loader").hide();
       }
     });
    //     $.ajax({
    //         type: "GET",
    //         url: serviseBaseUrl + "/textSum",
    //         data: JSON.stringify(upload_data),
    //         dataType: "jsonp",
    //         contentType: "application/json; charset=utf-8",
    //         success: function(dataSample) {
    //                  $(".loader").hide();
    //                  alert(dataSample['doc_summary']);
    //             // $(".uploadPreview").empty();
    //             // $(".loader").hide();
    //             // // // for (var i = 0; i < dataSample.length; i++) {
    //             //     var documentBlock = "<div class='documentBlock'>";
    //             //     var docType = dataSample.doc_type == "text" ? "Text File" : dataSample.doc_type == "pdf" ? "Pdf File" : dataSample.doc_type == "passport" ? "Passport" : dataSample.doc_type == "dl" ? "Driving Licence" : dataSample.doc_type == "twelve" ? "12th" : dataSample.doc_type == "tenth" ? "10th" : dataSample.doc_type;
    //             //     documentBlock += '<fieldset><legend align="center">' + docType + '</legend><div class="left-side">';
    //             //     for (var key in dataSample) {
    //             //         key !== "doc_type" ? documentBlock += '<div><label><b>' + key + ': </b></label>' + dataSample[key] + '</div>' : "";
    //             //     }
    //             //     //var imgSrc = $("a.thumb[doc_type=" + dataSample[i].doc_type + "]").attr("href");
    //             //     documentBlock += "</div><div class='left-side'>"+ dataSample.doc_summary + "</div></fieldset></div>";
    //             //     $(".personal .uploadPreview").append(documentBlock);
    //             //     if (dataSample.doc_type == "passport" || dataSample.doc_type == "dl") {
    //             //         $(".personal .uploadPreview").append(documentBlock);
    //             //     } else if (dataSample.doc_type == "tenth" || dataSample.doc_type == "twelve" || dataSample.doc_type == "grad" || dataSample.doc_type == "pg") {
    //             //         $(".educational .uploadPreview").append(documentBlock);
    //             //     } else {
    //             //         $(".professional .uploadPreview").append(documentBlock);
    //             //     }
    //             // // }
    //         },
    //         error: function(e) {
    //             $(".loader").hide();
    //             alert("Unable to process.");
    //         }
    //     });
    // }, 1000);

}

$(function() {

    addBlankTemp();
    $(".upload-section,.preview-section").accordion({ "icons": null });
    $("#tabs").tabs();
    $("#tabs > div").height(window.innerHeight - 490);
    $(document).on("change", 'input[name="myfile"]', (function(event) {
        var currentElm = $(event.target);
        doc_name = event.target.files[0].name;
        $(currentElm).parent().prev().val(doc_name);
        $(".uploadPreview").empty();
        getBase64(event.target.files[0]);
        setTimeout(function() {
            $(currentElm).parents(".added-elem").find(".thumb").attr("download", doc_name).attr("href", tmppath);
        }, 100);
        //tmppath = URL.createObjectURL(event.target.files[0]);
        //alert(tmppath);
    }));

    $(document).on("click", ".uploadPreview img", function(e) {
        $(".modal").remove();
        var modal = "<div class='modal'><img src='" + $(this).attr("src") + "'/><span class='close' title='Close'>&times;</span></div>";
        $(this).parent().append(modal);
        $(".modal").css("top", ($(this).position().top - 30) + "px");
    });
    $(document).on("click", ".modal span", function(e) {
        $(".modal").remove();
    });
    // Get the <span> element that closes the modal

    $(document).on("change", 'select', (function(event) {
        $(event.target).siblings(".thumb").attr("doc_type", $(event.target).val());
    }));

});

// function addNewDocument(elem) {
//     var parentChooseElem = $(elem).parents(".choosefile");
//     var current_docList = [];
//     $(parentChooseElem).find("select").map(function(index, valueindex) {
//         current_docList.push($(valueindex).val());
//     });
//     var currentSection = $(parentChooseElem).attr("section-name");
//     var addElem = currentSection == "personal" ? addpersonalElm : currentSection == "edu" ? addEduElm : addProfElm;
//     if ($(elem).parents(".added-elem").find("select").val() == "-1") {
//         alert("Please select document type before.");
//     } else if ($(parentChooseElem).find(".added-elem").last().find("input[type=text]").val() == "") {
//         alert("Please upload document before.");
//     } else if (!$(parentChooseElem).parent().hasClass("professional-info") && $(parentChooseElem).find(".added-elem").length == 4) {
//         alert("Can not add any more document.")
//     } else {
//         $(parentChooseElem).find(".added-elem").find("select").attr("disabled", true);
//         $(parentChooseElem).append(addElem);
//         $(parentChooseElem).find("select").last().find("option").map(function(index, thisinst) {
//             if (isInArray(thisinst.value, current_docList)) {
//                 thisinst.setAttribute("disabled", true);
//             }
//         });
//     }
// }

function cancelupload() {
    $(".uploadPreview").empty();
    doc_type = [];
    $(".choosefile").empty();
    addBlankTemp();
}

function addBlankTemp() {
    $(".personal-info .choosefile").append(addpersonalElm);
}

function isInArray(value, array) {
    return array.indexOf(value) > -1;
}

function getBase64(file) {
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function() {
        tmppath = reader.result;
    };
    reader.onerror = function(error) {
        console.log('Error: ', error);
    };
}

// function removeDocument(elem) {
//     var contentlength = $(elem).parents(".choosefile").find(".added-elem").length;
//     if (contentlength == 1) {
//         return
//     } else if (contentlength == 2) {
//         var current_doc_elem = $(elem).parent().parent();
//         var docIndex = doc_type.indexOf($(current_doc_elem).find("select").val());
//         doc_type.splice(docIndex, 1);
//         $(current_doc_elem).siblings().find("select").removeAttr("disabled");
//
//         $(current_doc_elem).siblings().find("select option[disabled=true]").map(function(index, valueindex) {
//             $(valueindex).removeAttr("disabled");
//         })
//         $(current_doc_elem).remove();
//
//     } else {
//         var current_doc_type = $(elem).parents(".added-elem").find("select").val();
//         var docIndex = doc_type.indexOf(current_doc_type);
//         $(elem).parents(".choosefile").find("select").last().find("option").map(function(index, valueindex) {
//             if (current_doc_type == $(valueindex).val()) {
//                 $(valueindex).removeAttr("disabled");
//             }
//         });
//         doc_type.splice(docIndex, 1);
//         $(elem).parents(".added-elem").remove();
//     }
// }
