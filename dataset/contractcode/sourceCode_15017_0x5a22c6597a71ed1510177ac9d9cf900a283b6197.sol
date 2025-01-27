/**
 *Submitted for verification at Etherscan.io on 2017-09-18
*/

pragma solidity ^0.4.0;

/// @title Andxor hash logger
/// @author Andxor Soluzioni Informatiche srl <http://www.andxor.it/>
contract AndxorLogger {
    event LogHash(uint256 hash);

    function AndxorLogger() {
    }

    /// logs an hash value into the blockchain
    function logHash(uint256 value) {
        LogHash(value);
    }
}