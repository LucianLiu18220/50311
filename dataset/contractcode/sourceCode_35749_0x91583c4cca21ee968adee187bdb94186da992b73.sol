/**
 *Submitted for verification at Etherscan.io on 2017-10-19
*/

pragma solidity ^0.4.0;

contract TestRevert {
    function test_require() public {
        require(now < 1000);
    }

    function test_assert() public {
        assert(now < 1000);
    }
}