
pragma solidity ^0.4.24;

contract VulnerableContract {
    uint public a;
    uint public b;
    uint public c;

    function setc() public returns (uint) {
        if (block.timestamp % 2 == 0) {
            a = 32767;
        }
        sumAB();
        return c;
    }
    function sumAB() public {
        c = b + a;
    }
}


