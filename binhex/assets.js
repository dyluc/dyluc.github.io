imgs = [];
drawChar = null;
_assetsLoaded = 0;



window.onload = function() {
    _srcs = ['imgs/grassTile.png', 'imgs/dirtTile.png', 'imgs/backgroundTile.png', 'imgs/cloudsTile.png'];
    //^^ make class object to extract from atlas ^^

    for(var i=0;i<_srcs.length;i++) {
        imgs.push({
            _img: new Image()
        })
        imgs[i]._img.src = _srcs[i];
        _assetsLoaded++;

    }
    
    //deal with characters
    //104x24
    //characters -> 4x5
    //gap 2px
    
    var chars = new Image();
    chars.onload = function() {
        var _self = this;
        var gap = 12;
        var wid = 24, hei = 30;
        var _hexChars = ['A', 'B', 'C', 'D', 'E', 'F'];
        
        drawChar = function(ctx, char, x, y, dw, dh, center) {//will autoCenter, intended for 1 character
            _characters = [];
            for(var num=0;num<16;num++) {                
                _characters.push({
                    _character: num <= 9 ? num.toString() : _hexChars[num-10],
                    srcImage: _self,
                    srcX: num <= 9 ? (wid+gap)*num : (wid+gap)*(num-10),
                    srcY: num <= 9 ? 0 : hei+gap,
                })
                
                
            }
            
            //if centet
            _offsetX = 0;
            _offsetY = 0;
            
            if(center) { //true
                _offsetX = dw/2;
                _offsetY = dh/2;
            }
            
            function findChar(element, index, array) {
                if(element._character === char.toString()) { //just incase is not a string
                    //draw center character
                    ctx.drawImage(element.srcImage, 
                                  element.srcX, 
                                  element.srcY, 
                                  wid, 
                                  hei, 
                                  x-_offsetX,
                                  y-_offsetY,
                                  dw, 
                                  dh);
                }
            }
            _characters.forEach(findChar);
            
            
        }
        _assetsLoaded++;
    }
    chars.src = 'imgs/characters.png';
}