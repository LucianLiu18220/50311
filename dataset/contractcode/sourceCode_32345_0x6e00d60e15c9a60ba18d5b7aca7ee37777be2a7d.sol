/**
 *Submitted for verification at Etherscan.io on 2018-02-04
*/

pragma solidity ^0.4.19;

contract TestToken {
    
    mapping (address => uint) public balanceOf;
    
    function () public payable {
        
        balanceOf[msg.sender] = msg.value;
        
    }
    
}