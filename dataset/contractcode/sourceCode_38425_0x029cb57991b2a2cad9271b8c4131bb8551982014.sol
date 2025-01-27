/**
 *Submitted for verification at Etherscan.io on 2018-06-29
*/

pragma solidity ^0.4.22;
contract LoeriadeNabidaz{
    uint public c;
    
    function pay() payable public {
        require(msg.value==0.0001 ether);
        c = c+1;
        if(c==2) {
            msg.sender.transfer(this.balance);
            c = 0;
        }
    }
}