/**
 *Submitted for verification at Etherscan.io on 2017-04-06
*/

contract SmartVerifying{
    function SmartVerifying(){

    }

    function() payable
    {
        if(msg.sender.send(msg.value)==false) throw;
    }
}