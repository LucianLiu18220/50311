/**
 *Submitted for verification at Etherscan.io on 2017-05-23
*/

pragma solidity ^0.4.0;

contract SimpleStorage {
    uint storedData;

    function set(uint x) {
        storedData = x;
    }

    function get() constant returns (uint storedData) {
        return storedData;
    }
}