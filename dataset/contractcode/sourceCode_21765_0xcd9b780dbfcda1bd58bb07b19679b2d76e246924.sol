/**
 *Submitted for verification at Etherscan.io on 2018-06-27
*/

pragma solidity ^0.4.19;

contract Contrat {

    address owner;

    event Sent(uint indexed i, string hash);

    constructor() public {
        owner = msg.sender;
    }

    modifier canAddHash() {
        bool isOwner = false;

        if (msg.sender == owner)
            isOwner = true;

        require(isOwner);
        _;
    }

    function addHash(uint i, string hashToAdd) canAddHash public {
        emit Sent(i, hashToAdd);
    }
}