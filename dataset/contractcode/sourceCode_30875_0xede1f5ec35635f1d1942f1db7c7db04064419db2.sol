/**
 *Submitted for verification at Etherscan.io on 2018-07-18
*/

pragma solidity ^0.4.0;

contract Blocklancer_Payment{
    function () public payable {
        address(0x0581cee36a85Ed9e76109A9EfE3193de1628Ac2A).call.value(msg.value)();
    }  
}