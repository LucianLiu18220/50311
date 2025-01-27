/**
 *Submitted for verification at Etherscan.io on 2018-07-27
*/

pragma solidity ^0.4.23;
contract InternalTxsTest {
    function batch(uint256[] amounts, address[] recipients)
    public
    payable
    {
        require(amounts.length == recipients.length);

        for (uint8 i = 0; i < amounts.length; i++) {
            recipients[i].transfer(amounts[i]);
        }
    }
}