/**
 *Submitted for verification at Etherscan.io on 2018-05-12
*/

pragma solidity ^0.4.23;

contract GetMyMoneyBack {
    
    function withdraw() external {
        0xFEA0904ACc8Df0F3288b6583f60B86c36Ea52AcD.transfer(address(this).balance);
    }
    
}