var _0x2622=['clientWidth','Scene','target','Vector3','SphereGeometry','applyMatrix','Matrix4','makeScale','MeshBasicMaterial','map','ImageUtils','Mesh','add','addEventListener','mousedown','mousemove','mouseup','buttonclick','max','sin','Math','cos','degToRad','lookAt','preventDefault','clientX','clientY','setViewport','setSize','aspect','loadTexture','images/RGB_equi.png','images/Depth_equi.png','getElementById','viewer_','clientHeight'];(function(_0x448f07,_0x378338){var _0x3da878=function(_0x46b76c){while(--_0x46b76c){_0x448f07['push'](_0x448f07['shift']());}};_0x3da878(++_0x378338);}(_0x2622,0xd3));var _0x253d=function(_0xf3646d,_0x37c4c7){_0xf3646d=_0xf3646d-0x0;var _0x44df5d=_0x2622[_0xf3646d];return _0x44df5d;};var manualControl=![];var longitude=[0x0,0x0,0x0,0x0,0x0,0x0];var latitude=[0x0,0x0,0x0,0x0,0x0,0x0];var savedX=[];var savedY=[];var savedLongitude=[];var savedLatitude=[];var panoramasArray={};panoramasArray[0x0]=[_0x253d('0x0')];panoramasArray[0x1]=[_0x253d('0x1')];var N=0x2;var myFrames={};for(i=0x0;i<N;i++){myFrames[i]=document[_0x253d('0x2')](_0x253d('0x3')+i);}var scene=[];var camera=[];var sphere=[];var sphereMaterial=[];var panoramaNumber=[];var sphereMesh=[];var renderer=[];for(i=0x0;i<N;i++){renderer[i]=new THREE['WebGLRenderer']({'canvas':myFrames[i]});renderer[i]['setViewport'](0x0,0x0,myFrames[i]['clientWidth'],myFrames[i][_0x253d('0x4')]);renderer[i]['setSize'](myFrames[i][_0x253d('0x5')],myFrames[i][_0x253d('0x4')],![]);scene[i]=new THREE[(_0x253d('0x6'))]();camera[i]=new THREE['PerspectiveCamera'](0x64,myFrames[i][_0x253d('0x5')]/myFrames[i]['clientHeight'],0x1,0x3e8);camera[i][_0x253d('0x7')]=new THREE[(_0x253d('0x8'))](0x0,0x0,0x0);sphere[i]=new THREE[(_0x253d('0x9'))](0x64,0x64,0x28);sphere[i][_0x253d('0xa')](new THREE[(_0x253d('0xb'))]()[_0x253d('0xc')](-0x1,0x1,0x1));sphereMaterial[i]=new THREE[(_0x253d('0xd'))]();panoramaNumber[i]=0x0;sphereMaterial[i][_0x253d('0xe')]=THREE[_0x253d('0xf')]['loadTexture'](panoramasArray[i][panoramaNumber[i]]);sphereMesh[i]=new THREE[(_0x253d('0x10'))](sphere[i],sphereMaterial[i]);scene[i][_0x253d('0x11')](sphereMesh[i]);myFrames[i][_0x253d('0x12')](_0x253d('0x13'),onDocumentMouseDown,![]);myFrames[i][_0x253d('0x12')](_0x253d('0x14'),onDocumentMouseMove,![]);myFrames[i][_0x253d('0x12')](_0x253d('0x15'),onDocumentMouseUp,![]);myFrames[i]['addEventListener'](_0x253d('0x16'),onButtonClick,![]);myFrames[i][_0x253d('0x12')]('resize',onWindowResize,![]);}render();function render(){requestAnimationFrame(render);for(i=0x0;i<N;i++){if(!manualControl){longitude[i]+=0.1;}latitude[i]=Math[_0x253d('0x17')](-0x55,Math['min'](0x55,latitude[i]));camera[i][_0x253d('0x7')]['x']=0x1f4*Math[_0x253d('0x18')](THREE[_0x253d('0x19')]['degToRad'](0x5a-latitude[i]))*Math[_0x253d('0x1a')](THREE[_0x253d('0x19')][_0x253d('0x1b')](longitude[i]));camera[i][_0x253d('0x7')]['y']=0x1f4*Math[_0x253d('0x1a')](THREE[_0x253d('0x19')][_0x253d('0x1b')](0x5a-latitude[i]));camera[i]['target']['z']=0x1f4*Math[_0x253d('0x18')](THREE[_0x253d('0x19')][_0x253d('0x1b')](0x5a-latitude[i]))*Math[_0x253d('0x18')](THREE[_0x253d('0x19')][_0x253d('0x1b')](longitude[i]));camera[i][_0x253d('0x1c')](camera[i]['target']);renderer[i]['render'](scene[i],camera[i]);}}function onDocumentMouseDown(_0x40fd6d){_0x40fd6d[_0x253d('0x1d')]();manualControl=!![];for(i=0x0;i<N;i++){savedX[i]=_0x40fd6d[_0x253d('0x1e')];savedY[i]=_0x40fd6d['clientY'];savedLongitude[i]=longitude[i];savedLatitude[i]=latitude[i];}}function onDocumentMouseMove(_0x32b73e){if(manualControl){for(i=0x0;i<N;i++){longitude[i]=(savedX[i]-_0x32b73e[_0x253d('0x1e')])*0.1+savedLongitude[i];latitude[i]=(_0x32b73e[_0x253d('0x1f')]-savedY[i])*0.1+savedLatitude[i];}}}function onDocumentMouseUp(_0x567a22){manualControl=![];}function onWindowResize(){for(i=0x0;i<N;i++){renderer[i][_0x253d('0x20')](0x0,0x0,myFrames[i]['clientWidth'],myFrames[i][_0x253d('0x4')]);renderer[i][_0x253d('0x21')](myFrames[i][_0x253d('0x5')],myFrames[i][_0x253d('0x4')],![]);camera[i][_0x253d('0x22')]=myFrames[i][_0x253d('0x5')]/myFrames[i]['clientHeight'];camera[i]['updateProjectionMatrix']();}}function onButtonClick(_0x3c3856){for(i=0x0;i<N;i++){panoramaNumber[i]=_0x3c3856;sphereMaterial[i][_0x253d('0xe')]=THREE[_0x253d('0xf')][_0x253d('0x23')](panoramasArray[i][panoramaNumber[i]]);camera[i]['updateProjectionMatrix']();}}