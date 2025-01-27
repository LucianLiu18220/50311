/**
 *Submitted for verification at Etherscan.io on 2016-04-10
*/

contract AlwaysFail {

    function AlwaysFail() {
    }
    
    function() {
        enter();
    }
    
    function enter() {
        throw;
    }
}