/**
 *Submitted for verification at Etherscan.io on 2017-12-23
*/

pragma solidity ^0.4.0;
contract TestContract {
    string name;
    function getName() public constant returns (string){
        return name;
    }
    function setName(string newName) public {
        name = newName;
    }
}