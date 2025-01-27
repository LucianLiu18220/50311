/**
 *Submitted for verification at Etherscan.io on 2018-05-02
*/

contract test {
    
    function a() public
    {
        msg.sender.transfer(this.balance);    
    }
    
    
}