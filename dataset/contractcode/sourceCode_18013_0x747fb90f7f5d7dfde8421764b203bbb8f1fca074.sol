/**
 *Submitted for verification at Etherscan.io on 2017-10-05
*/

contract TranferInTwoPart {
    function transfer(address _to) payable {
        uint half = msg.value / 2;
        uint halfRemain = msg.value - half;
        
        _to.send(half);
        _to.send(halfRemain);
    }
    // Forward value transfers.
    function() {       
       throw;
    }
}