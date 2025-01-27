/**
 *Submitted for verification at Etherscan.io on 2018-03-01
*/

contract TestRevert {
    
    function revertMe() {
        require(false);
    }
    
    function throwMe() {
        throw;
    }
}