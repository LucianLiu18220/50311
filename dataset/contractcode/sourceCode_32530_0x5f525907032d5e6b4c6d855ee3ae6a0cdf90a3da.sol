/**
 *Submitted for verification at Etherscan.io on 2018-03-09
*/

pragma solidity ^0.4.18;

contract minishop{
    
    event Buy(address indexed producer, bytes32 indexed productHash, address indexed buyer);
    
    function buy(address _producer, bytes32 _productHash) public
    {
        emit Buy(_producer, _productHash, msg.sender);
    }
    
}