//Variables
var randomHexValue = '',
	rHD_Array = [], //deal with hex digits
	binaryValue = '',
	convertedToHexValue = '';

	//canvas
	hexBugs = null,
	WIDTH = null,
	HEIGHT = null,
	tick = 1,
    score = 0,
    hexDrawOnce = true,
	ticking = false,
	groundHeight = null,

	canvas = null,
	ctx = null,
    hexCanvas = null,
    hexCtx = null;

	/*DOM Stuffs*/

	//ONLY used for reset
	var $value = $('.value'), 
		$square = $('.square'),
		$hexReset = $('.hex-container');

    var $log = $('#log');


$('.square').on('click', function() {
    if(ticking == false) {
        ticking = true;
        score = 0;
    }
	var $div = $( this ), //div that was clicked on
		$divPosition = $div.index()+1, //divs index relative to parent. '+1' used for nth-child()
		$divValue = $('.value-container .value:nth-child('+($divPosition)+')'), //reference values using divPosition
		$valueContainer = $('.value-container'),
		$hexContainer = $('.hex-container');
		
		

	binaryValue = '', //both empties on every click but still available outside click handler
	convertedToHexValue = '';
	
	

	if($divValue.attr('data-value') === '1') {
		$div.children('img').attr('src', 'imgs/buttonUp.png');
        $divValue.attr('data-value', '0');
	} else if($divValue.attr('data-value') === '0'){
        $divValue.attr('data-value', '1');
		$div.children('img').attr('src', 'imgs/buttonDown.png');
	}
    
    //set to its correct image
    if($divValue.attr('data-value') === '0') {
        $divValue.children('img').attr('src', 'imgs/binZero.png');
    } else if($divValue.attr('data-value') === '1') {
        $divValue.children('img').attr('src', 'imgs/binOne.png');
    }

	for(var i=0; i<8;i++) { //increment binaryValue string with 8 characters
		binaryValue += ($valueContainer.children().eq(i).attr('data-value')).toString(); 
	}	

	convertedToHexValue+=binToHex(binaryValue.substring(0,4)); //hex conversion of binary value
	convertedToHexValue+=binToHex(binaryValue.substring(4,8));

    $hexContainer.attr('data-hex', convertedToHexValue);
    
    //draw in $hexContainer canvas data-hex
    if(drawChar) {
        var charOne = $hexContainer.attr('data-hex')[0].toString();
        var charTwo = $hexContainer.attr('data-hex')[1].toString();
        hexCtx.clearRect(0, 0, hexCanvas.width, hexCanvas.height);
        drawChar(hexCtx, charOne, 0, 0, hexCanvas.width/2-2, hexCanvas.height);
        drawChar(hexCtx, charTwo, hexCanvas.width/2+2, 0, hexCanvas.width/2-2, hexCanvas.height);
    }

    $log.html(binaryValue);
})


	/*Canvas Stuffs*/

hexBugs = {
	bugs: [],
    
    
    //character image metadata... src size = 24px, 30px
    characterWidth: 12,
    characterHeight: 15,
    characterGap: 3, //gap between characters

	createBug: function() {
		var speeds = [0.41, 0.29, 0.35];
		if(tick % 300 === 0) {
			this.bugs.push({
				x: null,
				y: null,
                dFG: 50, //distance from ground
                _width: this.characterWidth*2 + this.characterGap, //values for size of characters bounding box
                _height: this.characterHeight,
				velX: 0.15,//speeds[Math.floor(Math.random()*speeds.length)],
                velY: 0.2,
				hexValue: null
			});
			//init x and y positions - side property affects x and y pos
			function initPos(element, index, array) {
				var n = array[index];
                
				//Check if x and y property are null
				if(n.x === null || n.y === null) {
					//set y and x position
		  			n.y = groundHeight-n._height-n.dFG;
		  			n.x = -(n._width+this.characterGap*2); //compensate for container size
				}
				if(n.hexValue === null) { //give hex value if it has none
					randomHexValue = '';
					generateRandomHex();

					n.hexValue = randomHexValue;
				}
				
			}
			this.bugs.forEach(initPos, this); 
		}
	},

	update: function() {
		//create loop through bugs array - forEach array method
		function bugArray(element, index, array) {
		  	var n = array[index];
		  	//update x position using velocity
		  	n.x+=n.velX;
            n.y = n.y + 0.2*Math.cos(tick/20);
            //this.y = height - 280 + 5*Math.cos(frames/10);
            
            //bob up and down
            
            
		  	//splice off array if out of bounds
		  	if(n.x>WIDTH) {
		  		array.splice(0, array.length);
		  		$log.html('Game over, press any button to start over');
		  		randomHexValue = '';
		  		tick = 1; //1 above 0 so createBug() does not push {} in array
		  		ticking = false;

		  	}
		  	//check randomHexValue against hexConversion
		  	if(n.hexValue === convertedToHexValue) {
		  		array.splice(index, 1);
		  		//use $valueContainer to 'reset' to 0
                
                score++;
                setTimeout(function(){
                    $value.attr('data-value', '0');
                    //set correct image for value
                    $value.children('img').attr('src', 'imgs/binZero.png'); //on every click, set div value's img src to its binZero
                    $square.children('img').attr('src', 'imgs/buttonUp.png');
                    $hexReset.attr('data-hex', '00');
                    //draw 00 hex conversion
                    //draw in $hexContainer canvas data-hex
                    if(drawChar) {
                        var charOne = $hexReset.attr('data-hex')[0].toString();
                        var charTwo = $hexReset.attr('data-hex')[1].toString();
                        hexCtx.clearRect(0, 0, hexCanvas.width, hexCanvas.height);
                        drawChar(hexCtx, charOne, 0, 0, hexCanvas.width/2-2, hexCanvas.height);
                        drawChar(hexCtx, charTwo, hexCanvas.width/2+2, 0, hexCanvas.width/2-2, hexCanvas.height);

                    }
                }, 750);
		  	}

		}
		this.bugs.forEach(bugArray);
		


		
	},

	draw: function() {
		function bugArray(element, index, array) {
			var n = array[index];

            
            
            //
            if(drawChar) {
                drawChar(ctx, n.hexValue[0], n.x, n.y, this.characterWidth, this.characterHeight);
                drawChar(ctx, n.hexValue[1], n.x+this.characterWidth+this.characterGap, n.y, this.characterWidth, this.characterHeight);
            }
		}
		this.bugs.forEach(bugArray, this);
	}
}
//Game Functions
function main() {
	canvas = document.getElementById('game-canvas');
	ctx = canvas.getContext('2d');

	canvas.width = WIDTH = 320; //because sizes take priority in css, these are more of 'resolution'
	canvas.height = HEIGHT = 480;

    
    //move into sprite file
    
    /*atlas = new Image();
    atlas.onload = function() {
       initAssets(); 
    }
    atlas.src = '';*/
    groundHeight = HEIGHT/4*3;
    
    
    //initiate hexCanvas and draw two 00
    hexCanvas = document.getElementById('hexCanvas'),
    hexCtx = hexCanvas.getContext('2d');
    hexCanvas.width = 52;
    hexCanvas.height= 30;
    
    
	var run = function() {
        if(_assetsLoaded >= 5) {
		  update();
		  render();
        }
		window.requestAnimationFrame(run, canvas);
	}
	window.requestAnimationFrame(run, canvas);
    
    
}

function update() {
	if(ticking) {
		tick++;
	}
    
	hexBugs.createBug();
	hexBugs.update();
}

function render() {
    
    if(drawChar && hexDrawOnce) { //only draw two 00 once
        var charOne = $hexReset.attr('data-hex')[0].toString();
        var charTwo = $hexReset.attr('data-hex')[1].toString();
        drawChar(hexCtx, charOne, 0, 0, hexCanvas.width/2-2, hexCanvas.height);
        drawChar(hexCtx, charTwo, hexCanvas.width/2+2, 0, hexCanvas.width/2-2, hexCanvas.height);
        hexDrawOnce = false;
    }
    
    
    
    backgroundPattern = ctx.createPattern(imgs[2]._img, 'repeat');
    cloudsPattern = ctx.createPattern(imgs[3]._img, 'repeat');
    dirtPattern = ctx.createPattern(imgs[1]._img, 'repeat');
    
    ctx.fillStyle = backgroundPattern;
	ctx.fillRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = dirtPattern;
    ctx.fillRect(0, groundHeight+64, WIDTH, HEIGHT-(groundHeight+64));
    
    //background -> redo all sprites in seperate file!
    for(var x=0;x<WIDTH;x+=100) {
        ctx.beginPath();
        ctx.fillStyle = cloudsPattern;
        ctx.arc(x, groundHeight-50, 50, 0, 2*Math.PI);
        ctx.arc(x+50, groundHeight-50, 55, 0, 2*Math.PI);
        ctx.fill();
    }
    
    for(var x=0;x<WIDTH;x+=64) {
        ctx.drawImage(imgs[0]._img, x, groundHeight);
        
        ctx.fillStyle = '#131f4e';
        ctx.fillRect(x, groundHeight-50, 20, 50);
        ctx.fillRect(x+10, groundHeight-64, 20, 64);
        ctx.fillRect(x+20, groundHeight-30, 20, 30);
        ctx.fillRect(x+30, groundHeight-10, 20, 10);
        ctx.fillRect(x+50, groundHeight-25, 20, 25);
    }
    //-------
    
    
    
	hexBugs.draw();
    if(drawChar) {
        x_spacing = 20;
                    
        for (var i = score.toString().length; i > 0; i--) {
            //i-1 -> index of characteer in string

           var c_n = score.toString()[i-1];

            drawChar(ctx, c_n, i*x_spacing, HEIGHT-20, 12, 15, true);
        }
    }
    
    
    
    

}

//Other functions
function generateRandomHex() {
	for(var b=0;b<2;b++) {
		rHD_Array[b] = Math.floor(Math.random() * (15 - 0 + 1)) + 0;

		if(rHD_Array[b] >= 10) {
			rHD_Array[b] = hexLetterConversion(rHD_Array[b]);
		}

		randomHexValue+=(rHD_Array[b].toString());
	}
}

function hexLetterConversion(singleDigit) { //Convert >=10 digits to letters
	var hexLetterValues = ['A', 'B', 'C', 'D', 'E', 'F'], //starting from 10
		digitToString = singleDigit.toString(), //convert to string
		hexIndex = digitToString[1], //reference index[1];
		digitConversion = hexLetterValues[hexIndex]; //use reference to index array
	
	return digitConversion;
}

function binToHex(stringSequence) { //for 4 character binary sequence
	var count = null,
		n_pos = null,
		val = null,
		n = [8, 4, 2, 1];
		
	for(var x=0;x<4;x++) {
		n_pos = n[x];
		val = stringSequence[x]*n_pos;
		count += val;
	}
	
	if(count>=10) {
		count = hexLetterConversion(count); //return value of hexLetterConversion().
	}
	return count;
}

main();


